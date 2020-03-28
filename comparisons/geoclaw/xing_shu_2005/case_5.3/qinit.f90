! qinit routine for parabolic bowl problem, only single layer
subroutine qinit(meqn,mbc,mx,my,xlower,ylower,dx,dy,q,maux,aux)

    use geoclaw_module, only: grav

    implicit none

    ! Subroutine arguments
    integer, intent(in) :: meqn,mbc,mx,my,maux
    real(kind=8), intent(in) :: xlower,ylower,dx,dy
    real(kind=8), intent(inout) :: q(meqn,1-mbc:mx+mbc,1-mbc:my+mbc)
    real(kind=8), intent(inout) :: aux(maux,1-mbc:mx+mbc,1-mbc:my+mbc)

    ! Parameters for problem
    real(kind=8), parameter :: eps = 1d-2
    real(kind=8), parameter :: w = 1d0

    ! Other storage
    integer :: i, j
    real(kind=8) :: x, y, eta

    do i=1-mbc, mx+mbc
        x = xlower + (i - 5d-1) * dx
        do j=1-mbc, my+mbc
            y = ylower + (j - 5d-1) * dy

            if ((x .ge. 0.05) .and. (x .le. 0.15)) then
                q(1, i, j) = w - aux(1, i, j) + eps
            else
                q(1, i, j) = w - aux(1, i, j)
            endif

            q(2, i, j) = 0d0
            q(3, i, j) = 0d0
        enddo
    enddo
    
end subroutine qinit
